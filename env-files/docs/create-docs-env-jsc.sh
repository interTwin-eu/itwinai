#!/bin/bash

# --------------------------------------------------------------------------------------
# Part of the interTwin Project: https://www.intertwin.eu/
#
# Created by: Matteo Bunino
#
# Credit:
# - Matteo Bunino <matteo.bunino@cern.ch> - CERN
# --------------------------------------------------------------------------------------

# Create .venv-docs virtualenv to build the documentation locally on JSC systems

ml --force purge
ml Stages/2023  GCCcore/.11.3.0 Python/3.10.4 Pandoc/2.19.2

cmake --version
gcc --version

rm -rf .venv-docs
python -m venv .venv-docs
source .venv-docs/bin/activate

pip install -r docs/requirements.txt