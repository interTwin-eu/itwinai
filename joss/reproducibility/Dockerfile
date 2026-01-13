# --------------------------------------------------------------------------------------
# Part of the interTwin Project: https://www.intertwin.eu/
#
# Created by: Matteo Bunino
#
# Credit:
# - Matteo Bunino <matteo.bunino@cern.ch> - CERN
# --------------------------------------------------------------------------------------

FROM ghcr.io/intertwin-eu/itwinai:0.3.4-slim-torch2.6-noble@sha256:0e25ba06f34bc7a9dbe4680933cb3e7a768fea5f2b8d2e1eb49a6cc20559928b
WORKDIR /experiments
COPY requirements.txt /experiments/
RUN uv pip install --no-cache-dir -r requirements.txt

# Ship experiment files inside the container 
COPY synthetic-data-gen /experiments/synthetic-data-gen
COPY src /experiments/src
COPY *.yaml *.py *.md Dockerfile /experiments/

ENV HYDRA_FULL_ERROR=1 

# Sample commands:
# docker build -t ghcr.io/intertwin-eu/itwinai:joss-virgo-experiments .
# docker push ghcr.io/intertwin-eu/itwinai:joss-virgo-experiments
