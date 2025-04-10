# --------------------------------------------------------------------------------------
# Part of the interTwin Project: https://www.intertwin.eu/
#
# Created by: Matteo Bunino
#
# Credit:
# - Matteo Bunino <matteo.bunino@cern.ch> - CERN
# --------------------------------------------------------------------------------------

FROM rocm/pytorch:rocm6.3.4_ubuntu22.04_py3.10_pytorch_release_2.4.0

WORKDIR /app
COPY pyproject.toml pyproject.toml
COPY src src
RUN pip install --no-cache-dir .[torch]