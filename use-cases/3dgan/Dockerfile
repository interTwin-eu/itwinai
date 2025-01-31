# --------------------------------------------------------------------------------------
# Part of the interTwin Project: https://www.intertwin.eu/
#
# Created by: Matteo Bunino
#
# Credit:
# - Matteo Bunino <matteo.bunino@cern.ch> - CERN
# --------------------------------------------------------------------------------------

FROM nvcr.io/nvidia/pytorch:23.09-py3

WORKDIR /usr/src/app

# Install itwinai
COPY pyproject.toml ./
COPY src ./
RUN pip install --upgrade pip \
    && pip install --no-cache-dir lightning \
    && pip install --no-cache-dir .

# Add 3DGAN use case files and install additional requirements
COPY use-cases/3dgan/requirements.txt ./
COPY use-cases/3dgan/* ./
RUN pip install --no-cache-dir -r requirements.txt

# ENTRYPOINT [ "itwinai", "exec-pipeline" ]
# CMD [ "--config", "pipeline.yaml" ]