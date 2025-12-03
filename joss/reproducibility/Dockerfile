FROM ghcr.io/intertwin-eu/itwinai:0.3.4-slim-torch2.6-noble@sha256:0e25ba06f34bc7a9dbe4680933cb3e7a768fea5f2b8d2e1eb49a6cc20559928b
WORKDIR /experiments
COPY requirements.txt /experiments/
RUN uv pip install --no-cache-dir -r requirements.txt
COPY . /experiments/
ENV HYDRA_FULL_ERROR=1 

# docker build -t ghcr.io/intertwin-eu/itwinai:joss-virgo-experiments .
# docker push ghcr.io/intertwin-eu/itwinai:joss-virgo-experiments
